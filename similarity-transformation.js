// umeyama_1991 refers to http://web.stanford.edu/class/cs273/refs/umeyama.pdf
//
// @article{umeyama_1991,
//   title={Least-squares estimation of transformation parameters between two point patterns},
//   author={Umeyama, Shinji},
//   journal={IEEE Transactions on Pattern Analysis \& Machine Intelligence},
//   number={4},
//   pages={376--380},
//   year={1991},
//   publisher={IEEE}
// }
//
// Variable names and the corresponding term in the paper's notation:
// - fromPoints refers to {x_i} with i = 1, 2, ..., n
// - toPoints refers to {y_i} with i = 1, 2, ..., n
// - numPoints refers to n
// - dimensions refers to m
// - rotation refers to R
// - scale refers to c
// - translation refers to t
// - fromMean and toMean refer to mu_x and mu_y respectively
// - fromVariance and toVariance refer to sigma_x and sigma_y respectively
// - mirrorIdentity refers to S
// - svd refers to the SVD given by U, D and V

function getSimilarityTransformationError(transformedPoints, toPoints) {
    // Computes equation 1 in umeyama_1991
    // Expects two mlMatrix.Matrix instances of the same shape (m, n), where n is the number of
    // points and m is the number of dimensions. This is the shape used by umeyama_1991.

    const numPoints = transformedPoints.columns;
    return (Math.pow(mlMatrix.Matrix.sub(toPoints, transformedPoints).norm('frobenius'), 2)
        / numPoints);
}

function getSimilarityTransformationErrorBound(fromPoints, toPoints, allowReflection = false) {
    // Computes equation 33 in umeyama_1991
    // Expects two mlMatrix.Matrix instances of the same shape (m, n), where n is the number of
    // points and m is the number of dimensions. This is the shape used by umeyama_1991.
    // The only restriction on the shape is that n and m are not zero.

    const dimensions = fromPoints.rows;

    // The variances will first be 1-D arrays and then reduced to a single number
    const summator = (sum, elem) => {
        return sum + elem;
    };
    const fromVariance = fromPoints.variance('row', {unbiased: false}).reduce(summator);
    const toVariance = toPoints.variance('row', {unbiased: false}).reduce(summator);
    const covarianceMatrix = getSimilarityTransformationCovariance(fromPoints, toPoints);

    const {
        svd,
        mirrorIdentityForErrorBound
    } = getSimilarityTransformationSvdWithMirrorIdentities(covarianceMatrix, allowReflection);

    let trace = 0;
    for (let dimension = 0; dimension < dimensions; dimension++) {
        trace += svd.diagonal[dimension] * mirrorIdentityForErrorBound[dimension];
    }
    return toVariance - Math.pow(trace, 2) / fromVariance;
}

function getSimilarityTransformation(fromPoints, toPoints, allowReflection = false) {
    // Computes equation 40, 41 and 42 in umeyama_1991
    // Expects two mlMatrix.Matrix instances of the same shape (m, n), where n is the number of
    // points and m is the number of dimensions. This is the shape used by umeyama_1991.
    // The only restriction on the shape is that n and m are not zero.

    const dimensions = fromPoints.rows;
    const numPoints = fromPoints.columns;

    // 1. Compute the rotation
    const covarianceMatrix = getSimilarityTransformationCovariance(fromPoints, toPoints);

    const {
        svd,
        mirrorIdentityForSolution
    } = getSimilarityTransformationSvdWithMirrorIdentities(covarianceMatrix, allowReflection);
    const rotation = svd.U
        .mmul(mlMatrix.Matrix.diag(mirrorIdentityForSolution))
        .mmul(svd.V.transpose());

    // 2. Compute the scale
    // The variance will first be a 1-D array and then reduced to a single number
    const summator = (sum, elem) => {
        return sum + elem;
    };
    const fromVariance = fromPoints.variance('row', {unbiased: false}).reduce(summator);

    let trace = 0;
    for (let dimension = 0; dimension < dimensions; dimension++) {
        trace += svd.diagonal[dimension] * mirrorIdentityForSolution[dimension];
    }
    const scale = trace / fromVariance;

    // 3. Compute the translation
    const fromMean = mlMatrix.Matrix.columnVector(fromPoints.mean('row'));
    const toMean = mlMatrix.Matrix.columnVector(toPoints.mean('row'));
    const translation = mlMatrix.Matrix.sub(
        toMean,
        mlMatrix.Matrix.mul(rotation.mmul(fromMean), scale));

    // 4. Transform the points
    const transformedPoints = mlMatrix.Matrix.add(
        mlMatrix.Matrix.mul(rotation.mmul(fromPoints), scale),
        translation.repeat({columns: numPoints}));

    return transformedPoints;
}

function getSimilarityTransformationCovariance(fromPoints, toPoints) {
    // Computes equation 38 in umeyama_1991
    // Expects two mlMatrix.Matrix instances of the same shape (m, n), where n is the number of
    // points and m is the number of dimensions. This is the shape used by umeyama_1991.
    // The only restriction on the shape is that n and m are not zero.

    const dimensions = fromPoints.rows;
    const numPoints = fromPoints.columns;
    const fromMean = mlMatrix.Matrix.columnVector(fromPoints.mean('row'));
    const toMean = mlMatrix.Matrix.columnVector(toPoints.mean('row'));

    const covariance = mlMatrix.Matrix.zeros(dimensions, dimensions);

    for (let pointIndex = 0; pointIndex < numPoints; pointIndex++) {
        const fromPoint = fromPoints.getColumnVector(pointIndex);
        const toPoint = toPoints.getColumnVector(pointIndex);
        const outer = mlMatrix.Matrix.sub(toPoint, toMean)
            .mmul(mlMatrix.Matrix.sub(fromPoint, fromMean).transpose());

        covariance.addM(mlMatrix.Matrix.div(outer, numPoints));
    }

    return covariance;
}

function getSimilarityTransformationSvdWithMirrorIdentities(covarianceMatrix, allowReflection) {
    const dimensions = covarianceMatrix.rows;
    const svd = new mlMatrix.SVD(covarianceMatrix);

    let mirrorIdentityForErrorBound = Array(svd.diagonal.length).fill(1);
    let mirrorIdentityForSolution = Array(svd.diagonal.length).fill(1);
    if (!allowReflection) {
        // Compute equation 39 in umeyama_1991
        if (mlMatrix.determinant(covarianceMatrix) < 0) {
            const lastIndex = mirrorIdentityForErrorBound.length - 1;
            mirrorIdentityForErrorBound[lastIndex] = -1;
        }

        // Check the rank condition mentioned directly after equation 43
        mirrorIdentityForSolution = mirrorIdentityForErrorBound;
        if (svd.rank === dimensions - 1) {
            // Compute equation 43 in umeyama_1991
            mirrorIdentityForSolution = Array(svd.diagonal.length).fill(1);
            if (mlMatrix.determinant(svd.U) * mlMatrix.determinant(svd.V) < 0) {
                const lastIndex = mirrorIdentityForSolution.length - 1;
                mirrorIdentityForSolution[lastIndex] = -1;
            }
        }
    }

    return {
        svd: svd,
        mirrorIdentityForErrorBound: mirrorIdentityForErrorBound,
        mirrorIdentityForSolution: mirrorIdentityForSolution
    }
}
